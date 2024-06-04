CC = gcc
CFLAGS = -O3 -DNDEBUG
DEBUGCFLAGS = -g -Wall -Wextra -DLOGGING
NAME = wcnfuzz

wcnfuzz: 
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME)

debug:
	$(CC) $(DEBUGCFLAGS) $(NAME).c -o $(NAME)

clean:
	rm $(NAME)
