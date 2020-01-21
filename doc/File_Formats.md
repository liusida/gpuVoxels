# File Formats used in Voxelyze3

`.vxa` and `.vxd` are the only two formats that an experimenter need to deal with.

* `.vxa` : Full specification for a simulation, adopted from Voxelyze 1.0; [Detail](https://github.com/liusida/gpuVoxels/blob/master/doc/VXA_File_Format.md)

* `.vxd` : Increamental parts or Override parts of a base `.vxa` file;

* `.vxt` : Task specification for a batch of `.vxa` files, including a base `.vxa`, a list of `.vxd`, and output path for `.vxr`;

* `.vxr` : Result for one batch of simulations, including the best fit, a list of sorted results;

* `.vxh` : Heart beatings for `vx3_node_daemon`'s, so we know which daemons are alive;

* `.vxc` : Reserved for old usage in Voxelyze 1.0 and 2.0;

