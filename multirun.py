import pypendula_n3
from tqdm import trange


for i in trange(25):
    ics = pypendula_n3.gen_rand_ics()
    pypendula_n3.main(ics=ics)
