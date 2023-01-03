# gcc -Wall:    Enable all warnings
# gcc -c:       Do not link
# gcc -g:       Add debug symbols
# gcc -lm:      Link to libm(ath)
# gcc -o:       Output file name
#
# Makefile $@:  The file name of the target of the rule
# Makefile $^:  The names of all the prerequisites

main: main.c liba.o libb.o
	gcc -g -Wall -o $@ $^ -lm

liba.o: liba.c
	gcc -g -Wall -c -o $@ $^

libb.o: libb.c
	gcc -g -Wall -c -o $@ $^

clean:
	rm -rf *.o main
