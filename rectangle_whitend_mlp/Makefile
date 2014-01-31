validation: validation.c
	gcc validation.c -Wall -Wextra -Ofast -fopenmp -lnv -o validation

submission: submission.c
	gcc submission.c -Wall -Wextra -Ofast -fopenmp -lnv -o submission

clean:
	rm -f validation
	rm -f submission

all: validation
