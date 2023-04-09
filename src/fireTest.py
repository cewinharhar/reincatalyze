import fire
import random

for i in range(100):
    oh = "".join([str(random.randint(a=0, b = 9)) for x in range(7)])
    nr = "+4176"+oh
    print(nr)




if __name__ == "__main__":
    fire.Fire()