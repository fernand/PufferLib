#include "td.h"
#include <stdlib.h>

int main() {
    int width = 30;
    int height = 30;
    int num_agents = 10;

    TDEnv env;
    init(&env, width, height, num_agents);

    env.observations = (float*)calloc(num_agents * TD_OBS_DIM, sizeof(float));
    env.actions      = (int*)calloc(num_agents, sizeof(int));
    env.rewards      = (float*)calloc(num_agents, sizeof(float));
    env.terminals    = (uint8_t*)calloc(num_agents, sizeof(uint8_t));
    env.truncations  = (uint8_t*)calloc(num_agents, sizeof(uint8_t));

    env.client = make_client(&env);

    c_reset(&env);
    c_render(&env);

    int agent_idx = 9;
    bool key_processed = false;
    while (!WindowShouldClose()) {
        // Handle input for the agent: one action per key press
        int new_action = TD_ACTION_NONE;
        if (IsKeyDown(KEY_W)     || IsKeyDown(KEY_UP))    new_action = TD_ACTION_UP;
        if (IsKeyDown(KEY_S)     || IsKeyDown(KEY_DOWN))  new_action = TD_ACTION_DOWN;
        if (IsKeyDown(KEY_A)     || IsKeyDown(KEY_LEFT))  new_action = TD_ACTION_LEFT;
        if (IsKeyDown(KEY_D)     || IsKeyDown(KEY_RIGHT)) new_action = TD_ACTION_RIGHT;
        int action = TD_ACTION_NONE;
        if (new_action != TD_ACTION_NONE) {
            if (!key_processed) {
                action = new_action;
                key_processed = true;
            }
        } else {
            key_processed = false;
        }
        env.actions[agent_idx] = action;

        // Step environment
        c_step(&env);
        // Reset if terminal
        if (env.terminals[agent_idx]) {
            c_reset(&env);
        }

        // Render updated state
        c_render(&env);
    }

    // Cleanup
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.truncations);
    close_client(env.client);
    c_close(&env);
    return 0;
}