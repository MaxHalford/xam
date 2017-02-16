from xam.tsa import util


sequence = 'appaaaaapapaapppaaapaaapapapppppuaapaapapaapaaaaaaaaaaappapppaaapaapaappaaaaaapaaaaaaaapaaapapaaappapapaapaaaappapappaapaaaaaaaappaaapaaaaapaapppappaaaaapaaaaaaappaa'

lengths = util.calc_subsequence_lengths(sequence)

# Calculate the average lengths
averages = {
    k: sum(v) / len(v)
    for k, v in lengths.items()
}

print(averages)
