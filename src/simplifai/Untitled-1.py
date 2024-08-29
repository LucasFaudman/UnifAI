

def make_poem() -> str:
    n = int(input("Enter number of lines: "))
    peom = ""
    for i in range(n):
        line = input(f"Enter line {i} or q to quit: ")
        if line == "q":
            break
        peom += line + "\n"
    return peom


if __name__ == "__main__":
    poem = make_poem()
    print(poem)






