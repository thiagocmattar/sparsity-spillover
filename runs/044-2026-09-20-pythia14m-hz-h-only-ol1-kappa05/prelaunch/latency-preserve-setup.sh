set -eu
control=/workspace/run044-control
root=/workspace/run044-latency
test -e "$root/runtime/precompile-ready"
test "$(cat "$control/compile-003.exit")" = 0
test "$(cat "$control/compile-004.exit")" = 0
for name in setup.log environment.exit compile-002.log compile-002-interruption.json compile-003.log compile-003.exit compile-004.py compile-004.log compile-004.exit; do
  test ! -e "$root/runtime/preparation-$name"
  cp "$control/$name" "$root/runtime/preparation-$name"
done
echo 'Preserved all preparation attempts for the final verified retrieval inventory'
