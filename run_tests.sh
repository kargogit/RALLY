./run_pipeline.sh tests/asm/v7.asm
./run_pipeline.sh tests/asm/v6.asm
./run_pipeline.sh tests/asm/v3.asm
./run_pipeline.sh tests/asm/v1.asm
./run_pipeline.sh tests/asm/golden.asm
./run_pipeline.sh tests/asm/golden_2.asm
./run_pipeline.sh tests/asm/structuse.asm
./run_pipeline.sh tests/asm/linklist.asm
./run_pipeline.sh tests/asm/kilo_15.asm
./run_pipeline.sh tests/asm/kill_15.asm
./run_pipeline.sh tests/asm/pwd_1.asm
./run_pipeline.sh tests/asm/pwd_11.asm
./run_pipeline.sh tests/asm/fannkuch_15.asm
./run_pipeline.sh tests/asm/fannkuch_18.asm
./run_pipeline.sh tests/asm/fannkuch_19.asm


lli v7_13.ll
lli v6_13.ll
lli -entry-function=_start v3_13.ll
lli -entry-function=_start v1_13.ll
lli golden_13.ll

gcc -c __gmon_start__.c -o __gmon_start__.o

clang -c -O0 -o structuse_13.o structuse_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out structuse_13.o __gmon_start__.o
./a.out

clang -c -O0 -o linklist_13.o linklist_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out linklist_13.o __gmon_start__.o
./a.out

clang -c -O0 -o kilo_15_13.o kilo_15_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out kilo_15_13.o __gmon_start__.o
./a.out

clang -c -O0 -o fannkuch_18_13.o fannkuch_18_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out fannkuch_18_13.o __gmon_start__.o
./a.out

clang -c -O0 -o kill_15_13.o kill_15_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out kill_15_13.o __gmon_start__.o
./a.out

clang -c -O0 -o pwd_1_13.o pwd_1_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out pwd_1_13.o __gmon_start__.o
./a.out

clang -c -O0 -o pwd_11_13.o pwd_11_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out pwd_11_13.o __gmon_start__.o
./a.out

clang -c -O0 -o fannkuch_19_13.o fannkuch_19_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out fannkuch_19_13.o __gmon_start__.o
./a.out

clang -c -O0 -o fannkuch_15_13.o fannkuch_15_13.ll
gcc -g -m64 -nostartfiles -O0 -o a.out fannkuch_15_13.o __gmon_start__.o
./a.out

lli -entry-function=_start golden_2_13.ll
