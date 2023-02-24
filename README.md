# Hubbard Magnetism

This program is used to calculate ground state energies of quantum dot plaquettes.
The program outputs a text file with LaTeX code which displays
tables of energies for given plaquettes, with ground-state energies written in bold.
See [Phys. Rev. B 107, 014403 (2023)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.014403) for more information.

## Run

To run, call `main(int argc, char** argv)`. This method passes a `vector<int> indeces` to `Main.run()`, which indicates the plaquettes to run.
Additional plaquettes can be added by appending them to `Main.plaquettes` and by changing the `indeces` appropriately.

## Author

Donovan Buterakos

## License

[MIT License](https://opensource.org/license/mit/)
