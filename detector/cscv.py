import numpy as np

from itertools import combinations
from detector.synthetic import daily_returns
from detector.stats import sharpe_ratio
from scipy.stats import rankdata

def split_into_blocks(returns, number_of_pieces): # cut the days into equal blocks
    return np.array_split(returns, number_of_pieces, axis=0)

def all_splits(num_blocks): # list every way to choose half the blocks
    block_numbers = range(num_blocks)
    half = num_blocks // 2

    return list(combinations(block_numbers, half))

def join_blocks(blocks, split): # glue chosen blocks back into the matrix

    chosen_blocks = []
    for block_number in split:
        chosen_blocks.append(blocks[block_number])

    return np.concatenate(chosen_blocks, axis=0)

def out_of_sample_blocks(num_blocks, split):
    """Block numbers that are NOT in the split."""

    leftover = []

    for block_number in range(num_blocks):
        if block_number not in split:
            leftover.append(block_number)

    return leftover
        
def logit_for_one_split(blocks, split): # pick the best on one half, see how it ranks on the other

    # glue the in-sample blocks into one matrix
    in_sample = join_blocks(blocks, split)

    # work out which blocks are out of sample
    out_blocks = out_of_sample_blocks(len(blocks), split)

    # glue the out of sample blocks into one matrix
    out_sample = join_blocks(blocks, out_blocks)

    # daily sharpe of every strategy, in sample
    in_sample_sharpes = sharpe_ratio(in_sample, annualise=False)

    # daily sharpe of every strategy, out of sample
    out_sample_sharpes = sharpe_ratio(out_sample, annualise=False)

    # find which strategy was best in sample
    best_index = np.nanargmax(in_sample_sharpes)


    out_sample_ranks = rankdata(out_sample_sharpes)
    best_rank = out_sample_ranks[best_index]

    omega = best_rank / (len(out_sample_sharpes) + 1)
    logit = np.log(omega / (1 - omega))

    return logit, in_sample_sharpes[best_index], out_sample_sharpes[best_index]


    # test

    #print("In-sample best strategy:", best_index)
    #print("Its in-sample Sharpe:", in_sample_sharpes[best_index])
    #print("Its out-of-sample Sharpe:", out_sample_sharpes[best_index])
    #print("Its out-of-sample rank:", best_rank, "out of", len(out_sample_sharpes))



def probability_of_backtest_overfitting(returns, num_blocks=8): # do logit_for...() on every split and count the failures

    if num_blocks % 2 != 0:
        raise ValueError("num_blocks must be even")

    blocks = split_into_blocks(returns, num_blocks)
    splits = all_splits(num_blocks)

    # logit, and the winner's in- and out-of-sample Sharpe, for every split
    logits = []
    winners_in_sample = []
    winners_out_sample = []

    for split in splits:
        logit, best_in, best_out = logit_for_one_split(blocks, split)
        logits.append(logit)
        winners_in_sample.append(best_in)
        winners_out_sample.append(best_out)

    num_below  = 0
    for logit in logits:
        if logit <= 0:
            num_below  += 1

    pbo = num_below  / len(logits)

    return pbo, logits, winners_in_sample, winners_out_sample



# testing
if __name__ == "__main__":
    returns = daily_returns(12, 3, seed=1)
    blocks = split_into_blocks(returns, 4)

    #print("Number of blocks:", len(blocks))
    #for block in blocks:
    #    print(block.shape)

    splits = all_splits(4)
    #print("Number of splits:", len(splits))
    #print(splits)

    #print("With 8 blocks:", len(all_splits(8)))

    returns = daily_returns(12, 3, seed=1)
    blocks = split_into_blocks(returns, 4)

    joined = join_blocks(blocks, (0, 2))
    #print(joined.shape)                              # (6, 3)
    #print(np.array_equal(joined[0:3], blocks[0]))    # True
    #print(np.array_equal(joined[3:6], blocks[2]))    # True

    #print(out_of_sample_blocks(4, (0, 2)))          # [1, 3]
    #print(out_of_sample_blocks(4, (1, 2)))          # [0, 3]
    #print(out_of_sample_blocks(8, (0, 3, 5, 6)))    # [1, 2, 4, 7]


    returns = daily_returns(252, 100, seed=42)
    blocks = split_into_blocks(returns, 8)
    logit_for_one_split(blocks, (0, 1, 2, 3))

    logit = logit_for_one_split(blocks, (0, 1, 2, 3))
    #print("Logit:", logit)


    returns = daily_returns(252, 100, seed=42)
    pbo, logits = probability_of_backtest_overfitting(returns, num_blocks=8)
    print("Number of splits:", len(logits))
    print("PBO:", round(pbo, 3))

    pbos = []
    for seed in range(20):
        returns = daily_returns(252, 100, seed=seed)
        pbo, logits = probability_of_backtest_overfitting(returns, num_blocks=8)
        pbos.append(pbo)

    print("Average PBO over 20 seeds:", round(np.mean(pbos), 3))