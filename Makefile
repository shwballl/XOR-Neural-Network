all: compile run

compile:
	g++ main.cpp -o main.exe 

run:
	main.exe