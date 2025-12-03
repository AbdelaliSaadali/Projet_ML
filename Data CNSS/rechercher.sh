#/bin/bash

fichier="$1"
valeur="$2"

while IFS= read -r ligne; do
    # Extraire la 2e colonne en utilisant le délimiteur ;
    deuxieme_col=$(echo "$ligne" | cut -d',' -f3)
    
    if [ "$deuxieme_col" = "$valeur" ]; then
        echo "$ligne"
    fi
done < "$fichier"
