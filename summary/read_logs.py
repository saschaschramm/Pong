from summary.logger import SummaryReader

summary_reader = SummaryReader("ppo_fc")
data = summary_reader.read()

for x, y in zip(data[0], data[1]):
    print("{0:} {1:.1f}".format(x,y))
