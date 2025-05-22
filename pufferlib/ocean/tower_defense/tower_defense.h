#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define TD_MAX_AGENTS 10
#define TD_MAX_TOWERS 5
#define MAX_TICK 75
#define TD_GAMMA 0.99f        // discount used in PBRS
#define TD_STEP_COST -0.001f  // small per‑step pressure

// Actions
#define TD_ACTION_NONE 0
#define TD_ACTION_UP 1
#define TD_ACTION_DOWN 2
#define TD_ACTION_LEFT 3
#define TD_ACTION_RIGHT 4

// Grid encoding
#define TD_EMPTY 0
#define TD_TOWER 1
#define TD_HOME 2
#define TD_AGENT_LOW 3
#define TD_AGENT_HIGH 4

#define TD_AGENT_LOW_HP 25
#define TD_AGENT_HIGH_HP 100
#define TD_HOME_HP 100
#define TD_AGENT_DMG 5
#define TD_TOWER_DMG 50
#define TD_TOWER_FIRE_RATE 3

struct Agent {
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
    // Exposed to RL backend (allocated by Python bindings)
    uint8_t *observations;
    int *actions;
    float *rewards;
    uint8_t *terminals;
    uint8_t *truncations;  // optional

    // Internal
    int width, height;
    int num_agents;
    int tick;
    Log log;

    float *returns;
    int *grid;
    int *prev_grid;
    struct Agent *agents;
    struct Tower towers[TD_MAX_TOWERS];
    struct Home home;

    struct Client *client;
} TDEnv;

static inline int clamp(int v, int mn, int mx) { return v < mn ? mn : (v > mx ? mx : v); }
static inline float clampf(float v, float mn, float mx) { return v < mn ? mn : (v > mx ? mx : v); }
static inline float potential(int x, int y, const TDEnv *env) {
    return -(float)(abs(x - env->home.x) + abs(y - env->home.y));
}

// Bresenham LOS
static bool check_los(const TDEnv *env, int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int cx = x0, cy = y0;
    while (cx != x1 || cy != y1) {
        int e2 = err << 1;
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

static void compute_observations(TDEnv *env) {
    int w = env->width, h = env->height, wh = w * h;
    int obs_dim = 4 * wh;
    for (int i = 0; i < env->num_agents; i++) {
        struct Agent *a = &env->agents[i];
        uint8_t *obs = &env->observations[i * obs_dim];
        if (!a->alive) {
            memset(obs, 0, obs_dim);
            continue;
        }
        // Channel 0 – home
        for (int idx = 0; idx < wh; idx++) obs[idx] = (env->grid[idx] == TD_HOME) ? 255 : 0;
        // Channel 1 – towers (128 idle / 255 fired this tick)
        for (int idx = 0; idx < wh; idx++) {
            if (env->grid[idx] == TD_TOWER) {
                int x = idx % w, y = idx / w;
                uint8_t v = 128;
                for (int t = 0; t < TD_MAX_TOWERS; t++) {
                    struct Tower *tw = &env->towers[t];
                    if (tw->range > 0 && tw->x == x && tw->y == y) {
                        if (tw->last_fired == env->tick) v = 255;
                        break;
                    }
                }
                obs[wh + idx] = v;
            } else
                obs[wh + idx] = 0;
        }
        // Channel 2 – self
        for (int idx = 0; idx < wh; idx++) {
            int x = idx % w, y = idx / w;
            obs[2 * wh + idx] = (a->x == x && a->y == y) ? 255 : 0;
        }
        // Channel 3 – HP map
        for (int idx = 0; idx < wh; idx++) {
            uint8_t v = 0;
            int x = idx % w, y = idx / w;
            for (int j = 0; j < env->num_agents; j++) {
                struct Agent *a2 = &env->agents[j];
                if (a2->alive && a2->x == x && a2->y == y) {
                    v = (uint8_t)a2->hp;
                    break;
                }
            }
            obs[3 * wh + idx] = v;
        }
    }
}

// -----------------------------------------------------------------------------
// Environment lifecycle helpers (init / reset / close)
// -----------------------------------------------------------------------------
void init(TDEnv *env) {
    env->grid = (int *)calloc(env->width * env->height, sizeof(int));
    env->prev_grid = (int *)calloc(env->width * env->height, sizeof(int));
    env->agents = (struct Agent *)calloc(env->num_agents, sizeof(struct Agent));
    env->returns = (float *)calloc(env->num_agents, sizeof(float));
    env->log = (Log){0};
}

static void zero_env_buffers(TDEnv *env) {
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(uint8_t));
    if (env->truncations) memset(env->truncations, 0, env->num_agents * sizeof(uint8_t));
    memset(env->returns, 0, env->num_agents * sizeof(float));
}

void c_reset(TDEnv *env) {
    env->tick = 0;
    zero_env_buffers(env);

    // Towers
    int tower_range = 1 + (env->width < env->height ? env->width : env->height) / 4;
    for (int t = 0; t < TD_MAX_TOWERS; t++) env->towers[t] = (struct Tower){0};
    env->towers[0] = (struct Tower){env->width / 2, env->height / 2, tower_range, -3};
    env->towers[1] = (struct Tower){env->width - 1, env->height / 2, tower_range, -3};
    env->towers[2] = (struct Tower){0, env->height / 2, tower_range, -3};

    // Home
    env->home = (struct Home){env->width / 2, 0, TD_HOME_HP, TD_HOME_HP};

    memset(env->grid, TD_EMPTY, env->width * env->height * sizeof(int));
    for (int t = 0; t < TD_MAX_TOWERS; t++) {
        if (env->towers[t].range > 0) {
            env->grid[env->towers[t].y * env->width + env->towers[t].x] = TD_TOWER;
        }
    }
    env->grid[env->home.y * env->width + env->home.x] = TD_HOME;

    // Agents
    for (int i = 0; i < env->num_agents; i++) {
        struct Agent *a = &env->agents[i];
        a->alive = true;
        a->max_hp = (i < env->num_agents / 2 ? TD_AGENT_LOW_HP : TD_AGENT_HIGH_HP);
        a->hp = a->max_hp;
        a->x = (i + 1) * env->width / (env->num_agents + 1);
        a->y = env->height - 1;
        env->grid[a->y * env->width + a->x] =
            (i < env->num_agents / 2 ? TD_AGENT_LOW : TD_AGENT_HIGH);
    }
    compute_observations(env);
}

void add_log(TDEnv *env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->log.perf += env->rewards[i] / env->num_agents;
        env->log.score += env->returns[i] / env->num_agents;
    }
    env->log.n++;
    env->log.episode_length += env->tick;
}

void c_close(TDEnv *env) {
    free(env->grid);
    free(env->prev_grid);
    free(env->agents);
    free(env->returns);
}

void c_step(TDEnv *env) {
    env->tick++;
    const float frac_damage = (float)TD_AGENT_DMG / (float)TD_HOME_HP;  // used below

    // Shared per‑step pressure
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] += TD_STEP_COST;
        env->returns[i] += TD_STEP_COST;
    }

    // Defeat via timeout
    if (env->tick > MAX_TICK) {
        float team_r = -1.0f;
        for (int i = 0; i < env->num_agents; i++) {
            env->terminals[i] = 1;
            env->rewards[i] += team_r;
            env->returns[i] += team_r;
        }
        add_log(env);
        c_reset(env);
        return;
    }

    // Victory
    if (env->home.hp <= 0) {
        float team_r = 1.0f;
        for (int i = 0; i < env->num_agents; i++) {
            env->terminals[i] = 1;
            env->rewards[i] += team_r;
            env->returns[i] += team_r;
        }
        add_log(env);
        c_reset(env);
        return;
    }

    memset(env->terminals, 0, env->num_agents * sizeof(uint8_t));
    memset(env->rewards, 0, env->num_agents * sizeof(float));  // reset then add step cost after
    // restore step cost (simpler to do here)
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] += TD_STEP_COST;
        env->returns[i] += TD_STEP_COST;
    }

    memcpy(env->prev_grid, env->grid, env->width * env->height * sizeof(int));
    for (int j = 0; j < env->num_agents; j++)
        if (!env->agents[j].alive)
            env->prev_grid[env->agents[j].y * env->width + env->agents[j].x] = TD_EMPTY;

    int num_alive = 0;
    // Move agents
    for (int i = 0; i < env->num_agents; i++) {
        struct Agent *a = &env->agents[i];
        if (!a->alive) continue;
        num_alive++;

        int old_x = a->x, old_y = a->y;
        int ax = old_x, ay = old_y;
        switch (env->actions[i]) {
            case TD_ACTION_UP:
                ay--;
                break;
            case TD_ACTION_DOWN:
                ay++;
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
        int dest_idx = ny * env->width + nx;
        if (nx != old_x || ny != old_y) occupied = (env->prev_grid[dest_idx] != TD_EMPTY);

        if (occupied) {
            env->rewards[i] += -0.1f;
            env->returns[i] += -0.1f;
            int code = (a->max_hp <= TD_AGENT_LOW_HP ? TD_AGENT_LOW : TD_AGENT_HIGH);
            env->grid[old_y * env->width + old_x] = code;
        } else {
            int code = (a->max_hp <= TD_AGENT_LOW_HP ? TD_AGENT_LOW : TD_AGENT_HIGH);
            env->grid[old_y * env->width + old_x] = TD_EMPTY;
            env->grid[ny * env->width + nx] = code;
            a->x = nx;
            a->y = ny;
            env->prev_grid[dest_idx] = TD_AGENT_HIGH;  // mark occupied for this step
        }

        // Potential‑based shaping (Ng+99)
        float phi_s = potential(old_x, old_y, env);
        float phi_sp = potential(a->x, a->y, env);
        float dense = TD_GAMMA * phi_sp - phi_s;
        env->rewards[i] += dense;
        env->returns[i] += dense;
    }

    // Defeat
    if (num_alive == 0) {
        add_log(env);
        c_reset(env);
        return;
    }

    // Contact damage
    for (int i = 0; i < env->num_agents; i++) {
        struct Agent *a = &env->agents[i];
        if (!a->alive) continue;
        if (abs(a->x - env->home.x) + abs(a->y - env->home.y) == 1) {
            env->home.hp -= TD_AGENT_DMG;
            env->rewards[i] += frac_damage;
            env->returns[i] += frac_damage;
        }
    }

    // Towers attack
    for (int t = 0; t < TD_MAX_TOWERS; t++) {
        struct Tower *tw = &env->towers[t];
        if (tw->range <= 0) continue;
        if (env->tick - tw->last_fired < TD_TOWER_FIRE_RATE) continue;
        int target = -1, range2 = tw->range * tw->range, best = range2 + 1;
        for (int i = 0; i < env->num_agents; i++) {
            struct Agent *a = &env->agents[i];
            if (!a->alive) continue;
            int dx = a->x - tw->x, dy = a->y - tw->y, dist2 = dx * dx + dy * dy;
            if (dist2 <= range2 && check_los(env, tw->x, tw->y, a->x, a->y))
                if (dist2 < best) {
                    best = dist2;
                    target = i;
                }
        }
        if (target >= 0) {
            tw->last_fired = env->tick;
            struct Agent *a = &env->agents[target];
            int hp_before = a->hp;
            a->hp -= TD_TOWER_DMG;
            if (a->hp <= 0) {
                float frac = (float)hp_before / (float)a->max_hp;
                float penalty = -0.25f * frac;
                a->alive = false;
                env->grid[a->y * env->width + a->x] = TD_EMPTY;
                env->terminals[target] = 1;
                env->rewards[target] += penalty;
                env->returns[target] += penalty;
            }
        }
    }

    // Reward clipping
    for (int i = 0; i < env->num_agents; i++)
        env->rewards[i] = clampf(env->rewards[i], -1.0f, 1.0f);

    compute_observations(env);
}

typedef struct Client {
    int px;
} Client;

static Client *make_client(TDEnv *env) {
    Client *c = (Client *)calloc(1, sizeof(Client));
    c->px = 32;
    InitWindow(env->width * c->px, env->height * c->px, "PufferLib TD (optimised)");
    SetTargetFPS(60);
    return c;
}

void c_render(TDEnv *env) {
    if (!env->client) env->client = make_client(env);
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    int px = env->client->px;
    for (int i = 0; i < env->height; i++)
        for (int j = 0; j < env->width; j++) {
            int code = env->grid[i * env->width + j];
            Color col;
            switch (code) {
                case TD_TOWER:
                    col = (Color){255, 0, 255, 255};
                    break;
                case TD_HOME:
                    col = (Color){0, 255, 0, 255};
                    break;
                case TD_AGENT_LOW:
                    col = (Color){255, 165, 0, 255};
                    break;
                case TD_AGENT_HIGH:
                    col = (Color){255, 0, 0, 255};
                    break;
                default:
                    continue;
            }
            DrawRectangle(j * px, i * px, px, px, col);
            if (code == TD_AGENT_LOW || code == TD_AGENT_HIGH) {
                for (int k = 0; k < env->num_agents; k++) {
                    struct Agent *e = &env->agents[k];
                    if (e->alive && e->x == j && e->y == i) {
                        const char *t = TextFormat("%d", e->hp);
                        int fs = px / 2;
                        DrawText(t, j * px + (px - MeasureText(t, fs)) / 2, i * px + (px - fs) / 2,
                                 fs, WHITE);
                        break;
                    }
                }
            }
            if (code == TD_HOME) {
                const char *t = TextFormat("%d", env->home.hp);
                int fs = px / 2;
                DrawText(t, j * px + (px - MeasureText(t, fs)) / 2, i * px + (px - fs) / 2, fs,
                         WHITE);
            }
        }
    EndDrawing();
}

void close_client(struct Client *c) {
    CloseWindow();
    free(c);
}
