
# Available Parameters:
advance          set indicator for advanced starting information
barrier          set parameters for barrier optimization
benders          set parameters for benders optimization
clocktype        set type of clock used to measure time
conflict         set parameters for finding conflicts
cpumask          set cpubinding mask (off, auto, or a hex mask)
defaults         set all parameter values to defaults
dettimelimit     set deterministic time limit in ticks
emphasis         set optimization emphasis
feasopt          set parameters for feasopt
logfile          set file to which results are printed
lpmethod         set method for linear optimization
mip              set parameters for mixed integer optimization
multiobjective   set parameters for multi-objective optimization
network          set parameters for network optimizations
optimalitytarget set type of solution CPLEX will attempt to compute
output           set extent and destinations of outputs
parallel         set parallel optimization mode
paramdisplay     set whether to display changed parameters before optimization
preprocessing    set parameters for preprocessing
qpmethod         set method for quadratic optimization
randomseed       set seed to initialize the random number generator
read             set problem read parameters
record           set record calls to C API
sifting          set parameters for sifting optimization
simplex          set parameters for primal and dual simplex optimizations
solutiontype     set solution information CPLEX will attempt to compute
threads          set default parallel thread count
timelimit        set time limit in seconds
tune             set parameters for parameter tuning
workdir          set directory for CPLEX working files
workmem          set memory available for working storage (in megabytes)

# MIP 強調の切り替え
値 記号 意味
0	CPX_MIPEMPHASIS_BALANCED	最適性と許容性のバランスを取ります (デフォルト)
1	CPX_MIPEMPHASIS_FEASIBILITY	最適性より許容性を重視します
2	CPX_MIPEMPHASIS_OPTIMALITY	許容性より最適性を重視します
3	CPX_MIPEMPHASIS_BESTBOUND	最適な境界の移動を重視します
4	CPX_MIPEMPHASIS_HIDDENFEAS	隠れた許容解を見つけることを重視します
5	CPX_MIPEMPHASIS_HEURISTIC	高品質の許容解を早く検出することを重視します
https://www.ibm.com/docs/ja/icos/22.1.2?topic=parameters-mip-emphasis-switch


# 数値精度の強調
値 ブール値 記号 対話式 意味
0	いいえ	CPX_OFF	いいえ	数値精度を重視しません (デフォルト)
1	はい	CPX_ON	はい	計算に細心の注意を払います


