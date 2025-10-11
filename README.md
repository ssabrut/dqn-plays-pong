# dqn-plays-pong


First problem, the model took long to learn because of the reward policy (stays -21 if the score is 0) and getting 1 for every score. The workaround is update the policy score to set the reward +5 when scoring, -5 when losing, and +0.01 when hitting the ball with the board.