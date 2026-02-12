import re

# Input and output file names
input_file = 'logs.txt'
output_file = 'retry_urls.txt'

# Regex explanation:
# Failed to download\s* -> Matches the prefix
# (https://.*?\.jpg)    -> Captures the URL starting with https and ending at .jpg
#                          .*? is "non-greedy" to stop at the first .jpg found
pattern = r"Failed to download\s*(https://.*?\.jpg)"

failed_links = []

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # findall handles cases where multiple "Failed to" logs 
            # are merged into a single line of text
            matches = re.findall(pattern, line)
            if matches:
                failed_links.extend(matches)

    # Remove duplicates if necessary
    unique_failed_links = list(set(failed_links))

    with open(output_file, 'a', encoding='utf-8') as f:
        for link in unique_failed_links:
            f.write(link + '\n')

    print(f"Extraction complete! Found {len(failed_links)} total failures.")
    print(f"Saved {len(unique_failed_links)} unique URLs to {output_file}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")