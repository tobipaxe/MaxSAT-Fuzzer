CC = g++
LIBS = -lz
CFLAGS = -std=c++17 -O3 -DNDEBUG
DEBUGCFLAGS = -std=c++17 -g -Wall -Wextra -DLOGGING
NAME = wcnfddmin

all:
	$(CC) $(CFLAGS) $(NAME).cpp -o $(NAME) $(LIBS)

debug:
	$(CC) $(DEBUGCFLAGS) $(NAME).cpp -o $(NAME) $(LIBS)

clean:
	rm $(NAME)
