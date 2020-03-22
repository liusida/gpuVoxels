This is the source code for preparing the Tuesday Meeting.

Gravity reduced, so voxels can be pushed into 3D shape.

Four different conditions:

1. [no constraints:]  Attach anywhere and anytime the cells touch.

2. [spatial constraints:]  Attach anytime but only when inside a bounding region, in the center of the floor.

3. [temporal constraints:]  Attach anywhere but only during windows in time, for example, no attachment 0-0.25 seconds, sticky 0.25-0.5 seconds,....

4. [spatiotemporal constraints:]  Attach only when inside bounding box and inside sticky time windows.