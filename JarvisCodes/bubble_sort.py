
def bubble_sort(data):
  """
  Sorts a list of numbers using the bubble sort algorithm.

  Args:
    data: A list of numbers to be sorted.

  Returns:
    A new list containing the sorted numbers.  The original list is not modified.
  """
  n = len(data)
  data_copy = data[:]  # Create a copy to avoid modifying the original list

  for i in range(n):
    # Outer loop: Iterate through the list n-1 times
    for j in range(0, n - i - 1):
      # Inner loop: Compare adjacent elements
      if data_copy[j] > data_copy[j + 1]:
        # Swap if the current element is greater than the next element
        data_copy[j], data_copy[j + 1] = data_copy[j + 1], data_copy[j]

  return data_copy

if __name__ == "__main__":
  # Example usage
  numbers = [64, 34, 25, 12, 22, 11, 33, 43, 22, 12]
  sorted_numbers = bubble_sort(numbers)

  print("Original list:", numbers)
  print("Sorted list:", sorted_numbers)


  #Another example
  numbers2 = [5, 1, 4, 2, 8]
  sorted_numbers2 = bubble_sort(numbers2)
  print("Original list:", numbers2)
  print("Sorted list:", sorted_numbers2)


##**Explanation:**

##1. **`bubble_sort(data)` Function:**
  ## - Takes a list `data` as input.
   ## `n = len(data)`: Gets the length of the list.
   ##data_copy = data[:]`: Creates a *copy* of the input list.  This is crucial – we don't want to modify the original list.  `[:]` is a slice that creates a shallow copy.
   #- **Outer Loop (`for i in range(n)`):**  This loop controls the number of passes through the list.  After each pass, the largest unsorted element "bubbles" to its correct position at