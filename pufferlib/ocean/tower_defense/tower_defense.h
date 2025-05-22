#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

// Forward declaration of rendering client
typedef struct Client Client;

#define TD_MAX_AGENTS 10
#define TD_MAX_TOWERS 5
#define MAX_TICK 50

#define TD_ACTION_NONE 0
#define TD_ACTION_UP 1
#define TD_ACTION_DOWN 2
#define TD_ACTION_LEFT 3
#define TD_ACTION_RIGHT 4

#define TD_EMPTY 0
#define TD_TOWER 1
#define TD_HOME 2
#define TD_ENEMY_LOW 3
#define TD_ENEMY_HIGH 4

#define TD_ENEMY_LOW_HP 25
#define TD_ENEMY_HIGH_HP 100
#define TD_HOME_HP 100

#define TD_AGENT_DMG 5
#define TD_TOWER_DMG 25
#define TD_TOWER_FIRE_RATE 3

struct Agents {
    int x, y;
    int hp;
    int max_hp;
    bool alive;
};
struct Tower {
    int x, y;
    int range;
    int last_fired;
};
struct Home {
    int x, y;
    int hp;
    int max_hp;
};

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    // RL-exposed buffers (set by Python, not allocated here)
    float *observations;   // size: num_agents * TD_OBS_DIM
    int *actions;          // size: num_agents
    float *rewards;        // size: num_agents
    float *returns;        // size: num_agents
    uint8_t *terminals;    // size: num_agents
    uint8_t *truncations;  // size: num_agents (optional, can be NULL)

    // Set by my_init() in binding.c
    int width;
    int height;
    int num_agents;

    int tick;
    Log log;
    Client *client;

    int *grid;
    int *prev_grid;
    struct Agents *agents;
    struct Tower towers[TD_MAX_TOWERS];
    struct Home home;
} TDEnv;

static void compute_observations(TDEnv *env) {
    int w = env->width;
    int h = env->height;
    int wh = w * h;
    int obs_dim = 4 * wh;
    for (int i = 0; i < env->num_agents; i++) {
        struct Agents *a = &env->agents[i];
        float *obs = &env->observations[i * obs_dim];
        if (!a->alive) {
            for (int j = 0; j < obs_dim; j++) obs[j] = 0.0f;
            continue;
        }
        // Channel 0: home
        for (int idx = 0; idx < wh; idx++) {
            obs[idx] = (env->grid[idx] == TD_HOME) ? 1.0f : 0.0f;
        }
        // Channel 1: towers (0.5 if not firing, 1.0 if firing)
        for (int idx = 0; idx < wh; idx++) {
            if (env->grid[idx] == TD_TOWER) {
                int x = idx % w;
                int y = idx / w;
                float v = 0.5f;
                for (int t = 0; t < TD_MAX_TOWERS; t++) {
                    struct Tower *tw = &env->towers[t];
                    if (tw->range > 0 && tw->x == x && tw->y == y) {
                        if (tw->last_fired == env->tick) v = 1.0f;
                        break;
                    }
                }
                obs[wh + idx] = v;
            } else {
                obs[wh + idx] = 0.0f;
            }
        }
        // Channel 2: self indicator
        for (int idx = 0; idx < wh; idx++) {
            int x = idx % w;
            int y = idx / w;
            obs[2 * wh + idx] = (a->x == x && a->y == y) ? 1.0f : 0.0f;
        }
        // Channel 3: normalized agent HP
        for (int idx = 0; idx < wh; idx++) {
            float v = 0.0f;
            int x = idx % w;
            int y = idx / w;
            for (int j = 0; j < env->num_agents; j++) {
                struct Agents *a2 = &env->agents[j];
                if (a2->alive && a2->x == x && a2->y == y) {
                    v = (float)a2->hp / (float)TD_ENEMY_HIGH_HP;
                    break;
                }
            }
            obs[3 * wh + idx] = v;
        }
    }
}

// Called by env_init via my_init: allocates internal state and sets up env
void init(TDEnv *env) {
    env->tick = 0;
    memset(&env->log, 0, sizeof(env->log));

    env->grid = (int *)calloc(env->width * env->height, sizeof(int));
    env->prev_grid = (int *)calloc(env->width * env->height, sizeof(int));
    env->agents = (struct Agents *)calloc(env->num_agents, sizeof(struct Agents));
    env->returns = (float *)calloc(env->num_agents, sizeof(float));

    // Place first tower at center, disable others
    for (int t = 0; t < TD_MAX_TOWERS; t++) {
        if (t == 0) {
            env->towers[t].x = env->width / 2;
            env->towers[t].y = env->height / 2;
            env->towers[t].range = (env->width < env->height ? env->width : env->height) / 4;
            env->towers[t].last_fired = -3;
        } else {
            env->towers[t].x = 0;
            env->towers[t].y = 0;
            env->towers[t].range = 0;
            env->towers[t].last_fired = -1;
        }
    }

    // Place home at bottom center
    env->home.x = env->width / 2;
    env->home.y = 0;
    env->home.max_hp = TD_HOME_HP;
    env->home.hp = env->home.max_hp;
}

// Reset environment state, but do not touch RL-exposed buffers (Python manages them)
void c_reset(TDEnv *env) {
    memset(env->grid, TD_EMPTY, env->width * env->height * sizeof(int));
    // Place all towers
    for (int t = 0; t < TD_MAX_TOWERS; t++) {
        if (env->towers[t].range > 0) {
            env->grid[env->towers[t].y * env->width + env->towers[t].x] = TD_TOWER;
        }
    }
    env->grid[env->home.y * env->width + env->home.x] = TD_HOME;
    env->home.hp = env->home.max_hp;

    for (int i = 0; i < env->num_agents; i++) {
        struct Agents *a = &env->agents[i];
        a->alive = true;
        // Initialize HP and max HP
        a->max_hp = (i < env->num_agents / 2 ? TD_ENEMY_LOW_HP : TD_ENEMY_HIGH_HP);
        a->hp = a->max_hp;
        a->x = (i + 1) * env->width / (env->num_agents + 1);
        a->y = env->height - 1;
        env->grid[a->y * env->width + a->x] = (i < env->num_agents / 2 ? TD_ENEMY_LOW : TD_ENEMY_HIGH);
        env->terminals[i] = 0;
    }
    memset(env->returns, 0, env->num_agents * sizeof(float));
    // Reset truncations buffer if provided
    if (env->truncations) {
        memset(env->truncations, 0, env->num_agents * sizeof(uint8_t));
    }
    compute_observations(env);
}

void add_log(TDEnv *env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->log.perf += env->rewards[i];
        env->log.score += env->returns[i];
        env->log.episode_length += env->tick;
        env->log.episode_return += env->returns[i];
        env->log.n++;
    }
}

static inline int clamp(int v, int mn, int mx) { return v < mn ? mn : (v > mx ? mx : v); }

// Bresenham line-of-sight between (x0,y0) and (x1,y1)
static bool check_los(const TDEnv *env, int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int cx = x0, cy = y0;
    while (cx != x1 || cy != y1) {
        int e2 = err * 2;
        if (e2 > -dy) {
            err -= dy;
            cx += sx;
        }
        if (e2 < dx) {
            err += dx;
            cy += sy;
        }
        if (cx == x1 && cy == y1) break;
        if (env->grid[cy * env->width + cx] != TD_EMPTY) return false;
    }
    return true;
}

// Step the environment by one tick. Actions are already in env->actions.
void c_step(TDEnv *env) {
    env->tick++;
    if (env->tick > MAX_TICK) {
        for (int i = 0; i < env->num_agents; i++) {
            env->terminals[i] = 1;
            env->returns[i] += -1.0f;
            env->rewards[i] += -1.0f;
            add_log(env);
            c_reset(env);
            return;
        }
    }
    if (env->home.hp <= 0) {
        for (int i = 0; i < env->num_agents; i++) {
            env->terminals[i] = 1;
            env->returns[i] += 1.0f;
            env->rewards[i] += 1.0f;
            add_log(env);
            c_reset(env);
            return;
        }
    }
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(uint8_t));
    int total = env->width * env->height;
    memcpy(env->prev_grid, env->grid, total * sizeof(int));
    // Remove dead agents from grid snapshot
    for (int j = 0; j < env->num_agents; j++) {
        struct Agents *a = &env->agents[j];
        if (!a->alive) {
            env->prev_grid[a->y * env->width + a->x] = TD_EMPTY;
        }
    }
    int num_alive = 0;
    for (int i = 0; i < env->num_agents; i++) {
        struct Agents *e = &env->agents[i];
        if (!e->alive) continue;
        num_alive++;
        int old_x = e->x, old_y = e->y;
        int ax = old_x, ay = old_y;
        switch (env->actions[i]) {
            case TD_ACTION_UP:
                ay++;
                break;
            case TD_ACTION_DOWN:
                ay--;
                break;
            case TD_ACTION_LEFT:
                ax--;
                break;
            case TD_ACTION_RIGHT:
                ax++;
                break;
            default:
                break;
        }
        int nx = clamp(ax, 0, env->width - 1);
        int ny = clamp(ay, 0, env->height - 1);
        bool occupied = false;
        if (nx != old_x || ny != old_y) {
            int code = env->prev_grid[ny * env->width + nx];
            if (code != TD_EMPTY) occupied = true;
        }
        if (occupied) {
            env->rewards[i] += -0.1f;
            env->returns[i] += -0.1f;
            // Remain in place and mark on grid
            int code = (e->max_hp <= TD_ENEMY_LOW_HP ? TD_ENEMY_LOW : TD_ENEMY_HIGH);
            env->grid[old_y * env->width + old_x] = code;
        } else {
            // Move and update grid
            int code = (e->max_hp <= TD_ENEMY_LOW_HP ? TD_ENEMY_LOW : TD_ENEMY_HIGH);
            env->grid[old_y * env->width + old_x] = TD_EMPTY;
            env->grid[ny * env->width + nx] = code;
            e->x = nx;
            e->y = ny;
        }
    }
    if (num_alive == 0) {
        add_log(env);
        c_reset(env);
        return;
    }
    // Home damage & reward
    for (int i = 0; i < env->num_agents; i++) {
        struct Agents *a = &env->agents[i];
        if (!a->alive) continue;
        int dx = abs(a->x - env->home.x);
        int dy = abs(a->y - env->home.y);
        if (dx + dy == 1) {
            env->home.hp -= TD_AGENT_DMG;
            env->returns[i] += 0.1f;
            env->rewards[i] += 0.1f;
        }
    }
    // Towers each select their closest in-range, line-of-sight enemy and fire once
    for (int t = 0; t < TD_MAX_TOWERS; t++) {
        struct Tower *tw = &env->towers[t];
        if (tw->range <= 0) continue;
        if (env->tick - tw->last_fired < TD_TOWER_FIRE_RATE) continue;
        int target = -1;
        int range2 = tw->range * tw->range;
        int best_dist2 = range2 + 1;
        for (int i = 0; i < env->num_agents; i++) {
            struct Agents *a = &env->agents[i];
            if (!a->alive) continue;
            int dx = a->x - tw->x;
            int dy = a->y - tw->y;
            int dist2 = dx * dx + dy * dy;
            if (dist2 <= range2 && check_los(env, tw->x, tw->y, a->x, a->y)) {
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    target = i;
                }
            }
        }
        if (target >= 0) {
            tw->last_fired = env->tick;
            struct Agents *a = &env->agents[target];
            a->hp -= TD_TOWER_DMG;
            if (a->hp <= 0) {
                a->alive = false;
                env->terminals[target] = 1;
                env->returns[target] += -1.0f;
                env->rewards[target] += -1.0f;
            }
        }
    }
    // Distance-based shaping reward: proportional to proximity to home
    {
        int max_dx = env->home.x > (env->width - 1 - env->home.x) ? env->home.x : (env->width - 1 - env->home.x);
        int max_dy = env->home.y > (env->height - 1 - env->home.y) ? env->home.y : (env->height - 1 - env->home.y);
        int max_dist = max_dx + max_dy;
        int denom = max_dist > 1 ? (max_dist - 1) : 1;
        for (int i = 0; i < env->num_agents; i++) {
            struct Agents *a = &env->agents[i];
            if (!a->alive) continue;
            int dx = abs(a->x - env->home.x);
            int dy = abs(a->y - env->home.y);
            int dist = dx + dy;
            float shaping = ((float)(max_dist - dist) / (float)denom) * 0.2f;
            if (dist <= 1) shaping = 0.2f;
            if (shaping < 0.0f) shaping = 0.0f;
            if (shaping > 0.2f) shaping = 0.2f;
            env->rewards[i] += shaping;
            env->returns[i] += shaping;
        }
    }
    compute_observations(env);
}

void c_close(TDEnv *env) {
    free(env->grid);
    free(env->prev_grid);
    free(env->agents);
}

struct Client {
    int px;
};

// Create rendering client (window, settings)
Client *make_client(TDEnv *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));
    int px = 32;
    InitWindow(env->width * px, env->height * px, "PufferLib TD");
    SetTargetFPS(60);
    client->px = px;
    return client;
}

void close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_render(TDEnv *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    int px = env->client->px;
    for (int i = 0; i < env->height; i++) {
        for (int j = 0; j < env->width; j++) {
            int code = env->grid[i * env->width + j];
            Color color;
            switch (code) {
                case TD_TOWER:
                    color = (Color){255, 0, 255, 255};
                    break;
                case TD_HOME:
                    color = (Color){0, 255, 0, 255};
                    break;
                case TD_ENEMY_LOW:
                    color = (Color){255, 165, 0, 255};
                    break;
                case TD_ENEMY_HIGH:
                    color = (Color){255, 0, 0, 255};
                    break;
                default:
                    continue;
            }
            DrawRectangle(j * px, i * px, px, px, color);
            // Display agent HP on their rectangle
            if (code == TD_ENEMY_LOW || code == TD_ENEMY_HIGH) {
                for (int k = 0; k < env->num_agents; k++) {
                    struct Agents *e = &env->agents[k];
                    if (e->alive && e->x == j && e->y == i) {
                        const char *hp_text = TextFormat("%d", e->hp);
                        int fontSize = px / 2;
                        int textWidth = MeasureText(hp_text, fontSize);
                        DrawText(hp_text, j * px + (px - textWidth) / 2, i * px + (px - fontSize) / 2, fontSize, WHITE);
                        break;
                    }
                }
            }
        }
    }
    EndDrawing();
}