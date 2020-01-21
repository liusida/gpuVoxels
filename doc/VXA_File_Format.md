This is a note for the format of configure files (VXA, VXC, etc...).

```bash
{Simulator}
    Integration
        Integrator              0                           //0 = euler in older versions
        DtFrac                  0.9                         //timestep % optimal dt
    Damping
        BondDampingZ            0.1                         //Physics Settings -> Other -> Bulk damping ratio (z)
        ColDampingZ             1.0                         //Collision damping ratio (z)
        SlowDampingZ            1.0                         //Ground damping ratio (z)
    Collisions
        SelfColEnabled          0                           //Enable Self Collision
        ColSystem               COL_SURFACE_HORIZON
        CollisionHorizon        2.0
    {Features}                                              //Not used
    StopCondition
        StopConditionType       0                           //Stop Condition. 2=Simulation Time
        StopConditionValue      0                           //Second
        InitCmTime              0                           //??? What is this? Center of Mass?
        ActuationStartTime      0                           //Not used
    EquilibriumMode
        EquilibriumModeEnabled  0                           //Not used
    {SurfMesh}                                              //Not related to backend
    MinTempFact                 0.1                         //only used in VX1: The smallest a voxel can go. (for stability reasons)
    {GA}                                                    //GA = Genetic Algorithm: Not related to backend, should be implement in Task Manager.

{Environment}
    Boundary_Conditions                                     //Boundary Conditions not used in Experiment
        NumBCs                  0
        {FRegion}
    /* in older version instead of {Boundary_Conditions}, there are two equivalents:
    {Fixed_Regions}
    {Forced_Regions}
    */
    Gravity
        GravEnabled             0                           //Physics Settings -> Environment -> Enable Gravity
        GravAcc                 -9.81                       //Gravity Value in m/s^2
        AlterGravityHalfway     1                           //Unknown
        FloorEnabled            0                           //Physics Settings ->Environment -> Enable Floor
        FloorSlope              0                           //!! This is interesting to floor change! allow from -89.0 to 89
    Thermal
        TempEnabled             0                           //Physics Settings -> Environment -> Enable Temperature
        TempBase                25                          //25 means no contraction
        TempAmplitude           0                           // = TempAmp - TempBase
        /* in older version, instead of TempAmpplitude, there is a equivalent:
        TempAmp                 25                          //The value under Enable Temperature in degree Celsius
        */
        VaryTempEnabled         0                           //Physics Settings -> Environment -> Vary Temperature
        TempPeriod              0.1                         //The value under Vary Temperature, Period, in seconds
    /* below are only used in VX1, ignored */
    {NeuralNet}
    {RegenerationModel}
    {ForwardModel}
    {Controller}
    {Signaling}
    {LightSource}
    GrowthAmplitude             0
    GrowthSpeedLimit            0
    GreedyGrowth                0
    GreedyThreshold             0
    TimeBetweenTraces           0
    SavePassiveData             0
    FluidEnvironment            0
    AggregateDragCoefficient    0
    FallingProhibited           0
    ContractionOnly             0
    ExpansionOnly               0
    DampEvolvedStiffness        0

{VXC}
    Lattice
        Lattice_Dim             0.001                       //Workspace -> Lattice Dim (meter here, mm in VoxCAD). It's just another term of "Voxel Size".
        X_Dim_Adj               1.0
        Y_Dim_Adj               1.0
        Z_Dim_Adj               1.0
        X_Line_Offset           0.0
        Y_Line_Offset           0.0
        X_Layer_Offset          0.0
        Y_Layer_Offset          0.0
    Voxel
        Vox_Name                "BOX" for VS_BOX            // VS_BOX, etc...
        X_Squeeze               1
        Y_Squeeze               1
        Z_Squeeze               1
    Palette
        {Material} x N
    Structure Compression="ASCII_READABLE"
        X_Voxels                1                           //Workspace->X Voxels (how many voxels in x-direction)
        Y_Voxels                1                           //Workspace->Y Voxels
        Z_Voxels                1                           //Workspace->Z Voxels
        Data                                                //
            Layer x N                                       //One layer of the robot. 0002000 stands for only one voxel with material 2 in the middle.
        PhaseOffset                                         //The structure is similar to <Data>, and this is using for make specific actuation.
            Layer x N                                       

        /* below can be ignored
        numForwardModelSynapses
        numControllerSynapses
        numRegenerationModelSynapses
        ControllerSynapseWeights
        ForwardModelSynapseWeights
        RegenerationModelSynapseWeights
        FinalPhaseOffset
        InitialVoxelSize
        FinalVoxelSize
        VestibularContribution
        PreDamageRoll
        PreDamagePitch
        PreDamageYaw
        StressContribution
        PreDamageStress
        PressureContribution
        PreDamagePressure
        Stiffness
        StressAdaptationRate
        PressureAdaptationRate
        */

{Material}
    MatType                     0                           //0 = SINGLE = Pallete->Material Type->"Basic", other values:[SINGLE, INTERNAL, DITHER, EXTERNAL]
    Name                        "Default"
    Display                                                 //Pallete -> Material Type -> Appearance
        Red                     0.5
        Green                   0.5
        Blue                    0.5
        Alpha                   1.0
    /* IF MatType==SINGLE */
        Mechanical
            MatModel            MDL_LINEAR                  //Pallete->Material Type->Model->Material Model
            {SSData}                                        //Not used
            Elastic_Mod         0                           //Pallete->Material Type->Model->Elastic Modulus (in Pa here, MPa in VoxCAD)
            Plastic_Mod         0                           //When MatModel=MDL_BILINEAR, Pallete->Material Type->Model->Plastic Modulus (in Pa here, MPa in VoxCAD)
            Yield_Stress        0                           
            Fail_Stress         0
            Fail_Strain         0
            Density             0                           //Pallete->Material Type->Physical->Density (kg/m^3)
            Poissons_Ratio      0                           //Pallete->Material Type->Physical->Poisson Ratio (usually 0 ~ 0.5)
            CTE                 0                           //Pallete->Material Type->Physical->CTE (Coefficient of Thermal Expansion) (This is for actuation while temperature vary enabled.)
            MaterialTempPhase   0                           //Pallete->Material Type->Physical->Temp Phase (What is this???)
            uStatic             0                           //Pallete->Material Type->Physical->Static Friction Coefficient
            uDynamic            0                           //Pallete->Material Type->Physical->Dynamic Friction Coefficient
            {FailModel}
    /* IF MatType==INTERNAL, DITHER, EXTERNAL */
        ...

{SurfMesh}                                                  //Not related to backend
    {CMesh}
        DrawSmooth                  0
        {Vertices}
        {Facets}
        {Lines}

{Features}                                                  //Not used
    MaxVelLimitEnabled      0.0                         
    MaxVoxVelLimit          0.1                         
    BlendingEnabled         0                           
    MixRadius
        x                   0
        y                   0
        z                   0
    BlendModel              MB_LINEAR
    PolyExp                 1.0
    FluidDampEnabled        0
    VolumeEffectsEnabled    0
    EnforceLatticeEnabled   0

{GA}                                                        //GA = Genetic Algorithm: Not related to backend, should be implement in Task Manager.
    Fitness                 0                           
    FitnessType             FT_NONE
    TrackVoxel              0
    FitnessFileName         ""
```

[Other formats](https://github.com/liusida/gpuVoxels/blob/master/doc/File_Formats.md)