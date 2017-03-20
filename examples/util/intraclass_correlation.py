from xam.util import intraclass_correlation

x = [1, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]
y = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']

print(intraclass_correlation(x, y))
