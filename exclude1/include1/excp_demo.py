class firstExp(Exception):
    pass
class invalid_filename(firstExp):
    pass
class blank(firstExp):
    pass
while True:
    try:
        name=str(input("enter filename"))
        if name!="":
            try:
                f=open(f"{name}"+".txt")
            except IOError:
                print("IOError")
            except Exception:
                print("Exception")
            else:
                print("file found")
                print(f.read())
                f.close()
            finally:
                print("finally printed")
        elif name is "":
            raise blank
        else:
            raise invalid_filename
    except invalid_filename:
        print("invalid filename")
    except blank:
        print("cannot be blank filename")
print("done")


def main():
    # Your script goes here
    a = dict()
    a[1]
    pass
import sys
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(type(e).__name__, e)
        type,value, traceback = sys.exc_info()
        print(type)
