pip install -r python-requirements.txt
antlr4 -Dlanguage=Python3 nasm_x86_64_lexer.g4
antlr4 -Dlanguage=Python3 nasm_x86_64_parser.g4
protoc --python_out=. ast.proto
protoc --cpp_out=. ast.proto
clang++-18 -std=c++17 -stdlib=libstdc++ `llvm-config-18 --cxxflags` llvmInit.cpp ast.pb.cc `llvm-config-18 --ldflags --libs core bitreader bitwriter support` -lprotobuf -o llvmInit
clang++-18 -std=c++17 -stdlib=libstdc++ `llvm-config-18 --cxxflags` llvmLowerInteger.cpp ast.pb.cc `llvm-config-18 --ldflags --libs core bitreader bitwriter support` -lprotobuf -o llvmLowerInteger
clang++-18 -std=c++17 -stdlib=libstdc++ `llvm-config-18 --cxxflags` llvmLowerFpAtomic.cpp ast.pb.cc `llvm-config-18 --ldflags --libs core bitreader bitwriter support` -lprotobuf -o llvmLowerFpAtomic
clang++-18 -std=c++17 -stdlib=libstdc++ `llvm-config-18 --cxxflags` llvmLowerControlFlow.cpp ast.pb.cc `llvm-config-18 --ldflags --libs core bitreader bitwriter support` -lprotobuf -o llvmLowerControlFlow
clang++-18 -std=c++17 -stdlib=libstdc++ `llvm-config-18 --cxxflags` llvmLowerVectorSSE.cpp ast.pb.cc `llvm-config-18 --ldflags --libs core bitreader bitwriter support` -lprotobuf -o llvmLowerVectorSSE
