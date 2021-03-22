def parse_log(filename, step_per_epoch=250):
    with open(filename + '.parsed.txt', 'w') as f:
        for line in open(filename):
            a, b, c = line.strip().split(' ')
            a = int(a)
            b = int(b)
            c = float(c)
            step = (a - 1) * step_per_epoch + b
            f.write('%d %f\n' % (step, c))


parse_log('worker.0.lbs-200+dbs-200-1616448809.txt')
parse_log('worker.0.lbs-200+dbs-50-1616448644.txt')
