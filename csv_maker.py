import random
import csv
import random


class FileReader:
    def __init__(self, filename):
        self.filename = filename
        self.line_gen = self.read_line_by_line()
        self.num_line = self.get_num_line()

    def read_line_by_line(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.strip()

    def get_next_line(self):
        try:
            return next(self.line_gen)
        except StopIteration:
            return None  # Or handle the end of file in some other way

    def get_num_line(self):
        f = open(self.filename, 'r', encoding='utf-8')
        return len(f.readlines())


class FileWriter:
    def __init__(self, filename):
        self.f = open(filename, 'w', newline='', encoding='utf-8')
        self.wr = csv.writer(self.f)
        self.wr.writerow(['label', 'context','ans'])

    def writer(self, conv, ans, label):
        self.wr.writerow([label, conv, ans])


fr = FileReader('reddit_conversations.4turns.test.txt')
fw = FileWriter('test.csv')
out_context = 'I feel very funny today.'
for i in range(fr.num_line):
    line = fr.get_next_line()
    line = line.split('\t')
    key = random.choice([0, 1])
    if key == 1:
        fw.writer(line[0] + line[1], line[2], key)

    else:
        fw.writer(line[0] + line[1], out_context, key)
        out_context = line[3]
