import json
import re
from pathlib import Path


def convertToJson(inputText):
    inputText = re.sub(r"//.*\n", "", inputText, flags=re.MULTILINE)

    # Step 1: Replace all ; with ,
    jsonText = inputText.replace(";", ",")

    # Step 2: Replace all = with :
    jsonText = jsonText.replace("=", ":")

    # Step 3: Remove , before } or ]
    jsonText = re.sub(r",(\s*[}\]])", r"\1", jsonText)

    # Step 4: Add quotes around keys and string values
    jsonText = re.sub(r"(\w+)\s*:", r'"\1":', jsonText)
    jsonText = re.sub(r":\s*(\w+)", r': "\1"', jsonText)
    jsonText = re.sub(r"\[\s*(\w+)", r'["\1"', jsonText)
    jsonText = re.sub(r",\s*(\w+)", r',"\1"', jsonText)
    jsonText = re.sub(r"(\w+)\s*:", r'"\1":', jsonText)

    # Step 5: Fix specific keys that should not be quoted as string values
    jsonText = re.sub(r":\s*\"(\d+)\"", r": \1", jsonText)

    # Step 6: Handle special arrays
    jsonText = re.sub(r'(\[\s*\n)(\s*")', r"\1", jsonText)
    jsonText = re.sub(r'(\s*"\])', r" \1", jsonText)

    return jsonText


def valueSubstitution(filePath):
    filePath = Path(filePath)
    with open(filePath, "r") as f:
        inputText = f.read()
        inputText
    valueDict = {}
    pattern = re.compile(r"(\w+)\s?=\s?(\d+)")
    substitutePattern = re.compile(r"(\w+)\s?=\s?(-?)\s?(\w+)")
    matches = pattern.findall(inputText)
    for key, value in matches:
        valueDict[key] = int(value)

    updatedInputText = re.sub(
        r"//.*\n", "", inputText, flags=re.MULTILINE
    )  # Remove comments

    def substitute_match(match):
        key = match.group(1)
        sign = match.group(2)
        value = match.group(3)
        # Check if the value is in resultsDict and replace if it is
        if value in valueDict:
            valueDict[key] = int(f"{sign}{valueDict[value]}")
            return f"{key} = {valueDict[key]}"
        else:
            return match.group(0)  # return the match unchanged

    cnt = 0
    while substitutePattern.search(updatedInputText) and cnt < 100:
        updatedInputText = substitutePattern.sub(substitute_match, updatedInputText)
        cnt += 1
    return updatedInputText


def cfg2json(filePath):
    filePath = Path(filePath)
    updatedInputText = valueSubstitution(filePath)
    outputText = convertToJson("{" + updatedInputText + "}")
    return outputText


def cfg2dict(filePath):
    return json.loads(cfg2json(filePath))


if __name__ == "__main__":
    # Path to fieldDimensions.cfg
    FieldDim=cfg2dict("<Path to BHuman/BadgerRLSystem repo root>/Config/Locations/Default/fieldDimensions.cfg")
    print(FieldDim)