# dqn-plays-pong


First problem, the model took long to learn because of the reward policy (stays -21 if the score is 0) and getting 1 for every score. The workaround is update the policy score to set the reward +5 when scoring, -5 when losing, and +0.01 when hitting the ball with the board.

Faced another issue starting at episode 153 till 167, where agent found a lazy strategy and stay still at the top or bottom of the arena, this is called local optimum, a strategy that is safe but ultimately ineffective. To fix this, i added a punishment for inactivity.

After the model is well trained (above 150 episode), the model moved to much which causing misses an easy ball. The solution is to hyperparameter tuning for `TARGET_UPDATE_FREQ`, `LEARNING_RATE` and the discount factor `GAMMA`.