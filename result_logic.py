def calculate_result(math_score, reading_score, writing_score):
    average = (
        math_score + reading_score + writing_score
    ) / 3

    if average >= 40:
        return "Pass"

    return "Needs Support"
