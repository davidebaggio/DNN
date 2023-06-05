CFLAGS= -Wall -Wextra -I./include/

run: compile
	./bin/DNN.exe

compile:
	g++ $(CFLAGS) ./src/*.cpp -o ./bin/DNN