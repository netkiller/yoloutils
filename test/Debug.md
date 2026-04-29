扩展名变小写

find . -type f -exec bash -c 'for f do b="${f%.*}";e="${f##*.}";[ "$b" != "$f" ] && [ "$e" != "${e,,}" ] && mv -n "$f" "$b.${e,,}";done' bash {} +