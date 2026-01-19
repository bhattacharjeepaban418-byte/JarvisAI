import random

print('Number Guessing Game')
num = random.randint(1, 10)
guess = int(input('Guess a number 1-10: '))
if guess == num:
    print('Correct!')
else:
    print(f'Wrong! It was {num}')
