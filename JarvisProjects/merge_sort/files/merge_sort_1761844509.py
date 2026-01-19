
def merge_sort(arr):
    """
    Implements the merge sort algorithm.

    Args:
        arr: The list to be sorted.

    Returns:
        The sorted list.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left_merge(left, right, [])
    right_merge(right, [])

    return merge(left, right, [])

def left_merge(left, right, arr):
    """
    Performs a left merge of two sorted lists.
    """
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[i] = left[i]
            i += 1
        else:
            arr[i] = right[j]
            j += 1
        i += 1

    while i < len(left):
        arr[i] = left[i]
        i += 1

    while j < len(right):
        arr[j] = right[j]
        j += 1

    return arr


def right_merge(right, arr):
    """
    Performs a right merge of two sorted lists.
    """
    i = 0
    j = 0
    while i < len(right):
        if right[i] <= arr[j]:
            arr[j] = right[i]
            j += 1
        else:
            arr[j] = right[i]
            j += 1
    return arr


def merge(left, right, arr):
    """
    Merges two sorted lists into a single sorted list.
    """
    i = 0
    j = 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j += 1
    return result


if __name__ == '__main__':
    arr = [12, 11, 13, 5, 6, 7]
    sorted_arr = merge_sort(arr)
    print(f"Sorted array: {sorted_arr}") #Output: Sorted array: [5, 6, 7, 11, 12, 13]
