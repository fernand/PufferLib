#include "td.h"
#include <stdlib.h>

int main() {
    int width = 20;
    int height = 20;
    int num_agents = 10;

    TDEnv env;
    init(&env, width, height, num_agents);

    env.observations = (float*)calloc(num_agents * TD_OBS_DIM, sizeof(float));
    env.actions      = (int*)calloc(num_agents, sizeof(int));
    env.rewards      = (float*)calloc(num_agents, sizeof(float));
    env.terminals    = (uint8_t*)calloc(num_agents, sizeof(uint8_t));
    env.truncations  = (uint8_t*)calloc(num_agents, sizeof(uint8_t));

    Client* client = make_client(&env);
    env.client = client;

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        // Handle input for the first agent
        env.actions[0] = TD_ACTION_NONE;
        if (IsKeyDown(KEY_W)     || IsKeyDown(KEY_UP))    env.actions[0] = TD_ACTION_UP;
        if (IsKeyDown(KEY_S)     || IsKeyDown(KEY_DOWN))  env.actions[0] = TD_ACTION_DOWN;
        if (IsKeyDown(KEY_A)     || IsKeyDown(KEY_LEFT))  env.actions[0] = TD_ACTION_LEFT;
        if (IsKeyDown(KEY_D)     || IsKeyDown(KEY_RIGHT)) env.actions[0] = TD_ACTION_RIGHT;

        // Step environment
        c_step(&env);
        // Reset if terminal
        if (env.terminals[0]) {
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
    close_client(client);
    c_close(&env);
    return 0;
}