#include "tower_defense.h"

int main() {
    TDEnv env;
    env.width = 30;
    env.height = 30;
    env.num_agents = 10;
    init(&env);

    env.observations = (float *)calloc(env.num_agents * TD_OBS_DIM, sizeof(float));
    env.actions = (int *)calloc(env.num_agents, sizeof(int));
    env.rewards = (float *)calloc(env.num_agents, sizeof(float));
    env.terminals = (uint8_t *)calloc(env.num_agents, sizeof(uint8_t));
    env.truncations = (uint8_t *)calloc(env.num_agents, sizeof(uint8_t));

    env.client = make_client(&env);

    c_reset(&env);
    c_render(&env);

    int agent_idx = 9;
    bool key_processed = false;
    while (!WindowShouldClose()) {
        // Handle input for the agent: one action per key press
        int new_action = TD_ACTION_NONE;
        if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) new_action = TD_ACTION_UP;
        if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) new_action = TD_ACTION_DOWN;
        if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) new_action = TD_ACTION_LEFT;
        if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) new_action = TD_ACTION_RIGHT;
        int action = TD_ACTION_NONE;
        if (new_action != TD_ACTION_NONE) {
            if (!key_processed) {
                action = new_action;
                key_processed = true;
                env.actions[agent_idx] = action;
                c_step(&env);
            }
        } else {
            key_processed = false;
            env.actions[agent_idx] = action;
        }

        if (env.terminals[agent_idx]) {
            c_reset(&env);
        }

        c_render(&env);
    }

    return 0;
}