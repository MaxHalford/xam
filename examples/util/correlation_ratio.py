from xam.util import correlation_ratio

x = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
y = [1, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]

print(correlation_ratio(x, y))
