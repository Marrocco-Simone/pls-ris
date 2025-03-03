#!/bin/bash

# Create thesis/imgs directory if it doesn't exist
mkdir -p ./thesis/imgs

# Copy images directory
cp -r ./paper/imgs/* ./thesis/imgs/

# Copy references.bib
cp ./paper/references.bib ./thesis/

# Create temporary file to store tex filenames
temp_file=$(mktemp)

# Copy tex files except main.tex and store filenames
for file in ./paper/*.tex; do
    filename=$(basename "$file")
    if [ "$filename" != "main.tex" ]; then
        cp "$file" "./thesis/"
        echo "$filename" >> "$temp_file"
    fi
done

# Clean start.tex if it already exists
if [ -f ./thesis/start.tex ]; then
    rm ./thesis/start.tex
fi

# Sort filenames and create start.tex with input commands
sort "$temp_file" | while read -r filename; do
    if [ "$filename" != "abstract.tex" ]; then
        echo "\\input{${filename%.*}}" >> ./thesis/start.tex
    fi
done

# Replace section commands in copied tex files
# for file in $(cat "$temp_file"); do
#     sed -i 's/\\section/\\chapter/g' "./thesis/$file"
#     sed -i 's/\\subsection/\\section/g' "./thesis/$file"
#     sed -i 's/\\subsubsection/\\subsection/g' "./thesis/$file"
# done

# Update .gitignore
echo "# Copied files" > ./thesis/.gitignore
echo "imgs/" >> ./thesis/.gitignore
echo "start.tex" >> ./thesis/.gitignore
echo ".gitignore" >> ./thesis/.gitignore
echo "references.bib" >> ./thesis/.gitignore
sort "$temp_file" | while read -r filename; do
    echo "$filename" >> ./thesis/.gitignore
done

# Remove temporary file
rm "$temp_file"

# cd ./thesis
# pdflatex marrocco_simone_computer_science_2024_2025.tex
# bibtex marrocco_simone_computer_science_2024_2025.aux
# pdflatex marrocco_simone_computer_science_2024_2025.tex
# pdflatex marrocco_simone_computer_science_2024_2025.tex