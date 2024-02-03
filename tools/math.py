def snap_to_power_of_2(a: int) -> int:
    if a <= 0:
        return 0

    # Find the highest power of 2 less than or equal to 'a'
    power = 0
    while (1 << power) <= a:
        power += 1

    # Calculate the difference between 'a' and the lower power of 2
    diff_lower = a - (1 << (power - 1))

    # Calculate the difference between the higher power of 2 and 'a'
    diff_higher = (1 << power) - a

    # Snap to the nearest power of 2
    if diff_lower < diff_higher:
        return 1 << (power - 1)
    else:
        return 1 << power
