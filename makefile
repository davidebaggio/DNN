CFLAGS= -Wall -Wextra -I./include/

compile:
	g++ $(CFLAGS) ./src/*.cpp -o ./bin/DNN
