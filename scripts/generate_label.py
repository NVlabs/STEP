
import os
import sys
import pickle

csv_path = sys.argv[1]
output_root = os.path.dirname(csv_path)
if 'train' in csv_path:
    output_file = os.path.join(output_root, 'train.pkl')
elif 'val' in csv_path:
    output_file = os.path.join(output_root, 'val.pkl')
else:
    raise ValueError(csv_path)

annots = {}
with open(csv_path, 'r') as fin:
    for line in fin:
        videoname, fid, x1,y1,x2,y2, label, pid = line.strip().split(',')
        fid = int(fid)
        box = [float(x1), float(y1), float(x2), float(y2)]
        label = int(label)
        pid = int(pid)

        if videoname in annots:
            if fid in annots[videoname]:
                if pid in annots[videoname][fid]:
                    annots[videoname][fid][pid]['label'].append(label)
                else:
                    annots[videoname][fid][pid] = {'label': [label],
                                                   'box': box}
            else:
                annots[videoname][fid] = {pid: {'label': [label],
                                                'box': box}}
        else:
            annots[videoname] = {fid: {pid: {'label': [label],
                                             'box': box}}}

with open(output_file, 'wb') as fout:
    pickle.dump(annots, fout)
