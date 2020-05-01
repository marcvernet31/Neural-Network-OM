clear;
% read the seeds in seeds.txt

[tr_seed, te_seed] = textread("seeds.txt", "%d %d");

% call function batch
uo_nn_batch(tr_seed,te_seed);
