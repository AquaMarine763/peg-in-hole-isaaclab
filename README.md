# Peg-in-Hole Isaac Lab

A clean Isaac Lab reproduction of a tutorial-style peg-in-hole local insertion task.

This project assumes coarse visual localization has already moved the peg near the hole. Reinforcement learning is responsible only for the final-stage local alignment and insertion. The current setup uses a UR10e arm, a fixed cylindrical peg, a grounded hole block, an external depth camera, wrist force/torque observations, and a two-phase policy with pre-contact visual alignment and post-contact insertion correction.

Current focus:

- keep the environment and task definition minimal and clean
- validate reset geometry, observations, rewards, and training stability
- iterate toward a robust local insertion benchmark before larger sim-to-real extensions
