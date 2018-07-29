from summary.logger import SummaryReader
summary_reader = SummaryReader("a2c")
summary_reader.plot(left_limit=2e4, right_limit=1e6, bottom_limit=-21, top_limit=21, save=True)