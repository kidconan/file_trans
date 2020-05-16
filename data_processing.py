import os
import csv
import pandas
import hyperparams as hp

with open(os.path.join(hp.data_path, '000001-010000.txt'), 'r', encoding='utf-8') as f:
    data = f.readlines()

amount = int(len(data)/2)

csvfile = open('./BZNSYP.csv', 'w+', newline="")
try:
    csv_writer = csv.writer(csvfile)
    for i in range(amount):
        name_line = data[2*i].strip()
        name_line = name_line.replace('\t', ' ').split(' ', maxsplit=1)[0]

        pin_yin = data[2*i+1].strip()

        txt = name_line + '|' + pin_yin
        
        csv_writer.writerow([txt])
except:
    print("Error in %d-th sample" % i)
finally:
    csvfile.close()
