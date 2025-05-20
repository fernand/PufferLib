#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Maximum number of agents included in the observation
#define TD_MAX_AGENTS 10

// Observation dimension:
// x, y, hp_norm,
// building_dx, building_dy, building_hp_norm,
// tower_dx, tower_dy,
// plus all agents' relative positions (dx, dy) and health (3 features per agent)
#define TD_OBS_DIM (8 + TD_MAX_AGENTS * 3)

// Action codes
#define TD_ACTION_NONE 0
#define TD_ACTION_UP 1
#define TD_ACTION_DOWN 2
#define TD_ACTION_LEFT 3
#define TD_ACTION_RIGHT 4

// Entity types for grid (optional, for LOS checks)
#define TD_EMPTY 0
#define TD_TOWER 1
#define TD_BUILDING 2
#define TD_ENEMY_LOW 3
#define TD_ENEMY_HIGH 4

// HP values
#define TD_ENEMY_LOW_HP 25
#define TD_ENEMY_HIGH_HP 100
#define TD_BUILDING_HP 100

// Damage values
#define TD_BUILDING_DMG 1  // per step when adjacent
#define TD_TOWER_DMG 1     // per step when in range

// Reward shaping (placeholder)
#define TD_REWARD_BUILDING_DAMAGE 1.0f
#define TD_REWARD_DEATH -1.0f

struct Enemy {
    int x, y;
    int hp;
    int max_hp;
    bool alive;
};
struct Tower {
    int x, y;
    int range;
};
struct Building {
    int x, y;
    int hp;
    int max_hp;
};

// Main environment struct, matching env_binding.h expectations
typedef struct {
    // RL-exposed buffers (set by Python, not allocated here)
    float *observations;   // size: num_enemies * TD_OBS_DIM
    int *actions;          // size: num_enemies
    float *rewards;        // size: num_enemies
    uint8_t *terminals;    // size: num_enemies
    uint8_t *truncations;  // size: num_enemies (optional, can be NULL)

    // Internal state
    int width;
    int height;
    int num_enemies;
    int max_enemies;

    int *grid;              // size: width*height, holds entity codes
    struct Enemy *enemies;  // size: max_enemies
    struct Tower tower;
    struct Building building;
} TDEnv;

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

// Called by env_init via my_init: allocates internal state and sets up env
void init(TDEnv *env, int width, int height, int num_enemies) {
    env->width = width;
    env->height = height;
    env->max_enemies = num_enemies;
    env->num_enemies = num_enemies;

    env->grid = (int *)calloc(width * height, sizeof(int));
    env->enemies = (struct Enemy *)calloc(env->max_enemies, sizeof(struct Enemy));

    // place tower at center
    env->tower.x = width / 2;
    env->tower.y = height / 2;
    env->tower.range = (width < height ? width : height) / 4;

    // place building at bottom center
    env->building.x = width / 2;
    env->building.y = 0;
    env->building.max_hp = TD_BUILDING_HP;
    env->building.hp = env->building.max_hp;
}

// Reset environment state, but do not touch RL-exposed buffers (Python manages them)
void c_reset(TDEnv *env) {
    memset(env->grid, TD_EMPTY, env->width * env->height * sizeof(int));
    env->grid[env->tower.y * env->width + env->tower.x] = TD_TOWER;
    env->grid[env->building.y * env->width + env->building.x] = TD_BUILDING;
    env->building.hp = env->building.max_hp;

    for (int i = 0; i < env->num_enemies; i++) {
        struct Enemy *e = &env->enemies[i];
        e->alive = true;
        // initialize HP and max HP
        e->max_hp = (i < env->num_enemies / 2 ? TD_ENEMY_LOW_HP : TD_ENEMY_HIGH_HP);
        e->hp = e->max_hp;
        e->x = (i + 1) * env->width / (env->num_enemies + 1);
        e->y = env->height - 1;
        env->grid[e->y * env->width + e->x] =
            (i < env->num_enemies / 2 ? TD_ENEMY_LOW : TD_ENEMY_HIGH);
        env->terminals[i] = 0;
    }
    memset(env->rewards, 0, env->num_enemies * sizeof(float));
    // reset truncations buffer if provided
    if (env->truncations) {
        memset(env->truncations, 0, env->num_enemies * sizeof(uint8_t));
    }
    // Observations will be filled in c_step
}

// Step the environment by one tick. Actions are already in env->actions.
void c_step(TDEnv *env) {
    // reset rewards
    memset(env->rewards, 0, env->num_enemies * sizeof(float));
    // clear grid except tower/building
    for (int i = 0; i < env->width * env->height; i++) {
        int code = env->grid[i];
        if (code == TD_TOWER || code == TD_BUILDING) continue;
        env->grid[i] = TD_EMPTY;
    }
    // move enemies
    for (int i = 0; i < env->num_enemies; i++) {
        struct Enemy *e = &env->enemies[i];
        if (!e->alive) continue;
        int ax = e->x, ay = e->y;
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
        e->x = clamp(ax, 0, env->width - 1);
        e->y = clamp(ay, 0, env->height - 1);
    }
    // re-place enemies in grid
    for (int i = 0; i < env->num_enemies; i++) {
        struct Enemy *e = &env->enemies[i];
        if (!e->alive) continue;
        env->grid[e->y * env->width + e->x] =
            (e->hp <= TD_ENEMY_LOW_HP ? TD_ENEMY_LOW : TD_ENEMY_HIGH);
    }
    // building damage & reward
    for (int i = 0; i < env->num_enemies; i++) {
        struct Enemy *e = &env->enemies[i];
        if (!e->alive) continue;
        int dx = abs(e->x - env->building.x);
        int dy = abs(e->y - env->building.y);
        if (dx + dy == 1) {
            env->building.hp -= TD_BUILDING_DMG;
            env->rewards[i] += TD_REWARD_BUILDING_DAMAGE;
        }
    }
    // tower damage & death
    for (int i = 0; i < env->num_enemies; i++) {
        struct Enemy *e = &env->enemies[i];
        if (!e->alive) continue;
        int dx = e->x - env->tower.x;
        int dy = e->y - env->tower.y;
        if (dx * dx + dy * dy <= env->tower.range * env->tower.range) {
            if (check_los(env, env->tower.x, env->tower.y, e->x, e->y)) {
                e->hp -= TD_TOWER_DMG;
                if (e->hp <= 0) {
                    e->alive = false;
                    env->terminals[i] = 1;
                    env->rewards[i] += TD_REWARD_DEATH;
                }
            }
        }
    }
    // compute observations
    for (int i = 0; i < env->num_enemies; i++) {
        struct Enemy *e = &env->enemies[i];
        float *obs = &env->observations[i * TD_OBS_DIM];
        if (!e->alive) {
            for (int j = 0; j < TD_OBS_DIM; j++) obs[j] = 0.0f;
            continue;
        }
        obs[0] = (float)e->x / (env->width - 1);
        obs[1] = (float)e->y / (env->height - 1);
        // normalized health
        obs[2] = (float)e->hp / (float)e->max_hp;
        obs[3] = (float)(env->building.x - e->x) / env->width;
        obs[4] = (float)(env->building.y - e->y) / env->height;
        obs[5] = (float)env->building.hp / env->building.max_hp;
        obs[6] = (float)(env->tower.x - e->x) / env->width;
        obs[7] = (float)(env->tower.y - e->y) / env->height;
        // include all agents' relative positions and normalized health
        for (int j = 0; j < TD_MAX_AGENTS; j++) {
            int idx_off = 8 + j * 3;
            if (j < env->num_enemies && env->enemies[j].alive) {
                struct Enemy *e2 = &env->enemies[j];
                float dx = (float)(e2->x - e->x) / env->width;
                float dy = (float)(e2->y - e->y) / env->height;
                // normalized health of each agent
                float hp_norm = (float)e2->hp / (float)e2->max_hp;
                obs[idx_off]     = dx;
                obs[idx_off + 1] = dy;
                obs[idx_off + 2] = hp_norm;
            } else {
                obs[idx_off] = obs[idx_off + 1] = obs[idx_off + 2] = 0.0f;
            }
        }
    }
}

// Optional: render and close functions (no-op if not used)
void c_render(TDEnv *env) {}
void c_close(TDEnv *env) {
    free(env->grid);
    free(env->enemies);
}