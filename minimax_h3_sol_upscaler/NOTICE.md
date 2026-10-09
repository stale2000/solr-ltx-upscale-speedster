## Upstream code

The `sol_refiner_h3/` package is copied from NVlabs/Sana's MiniMax-H3 SoL
Refiner at commit `670482d8a857d578ac8a2ea89b052d0fb47badba` and is licensed
under Apache-2.0. The license text is in `licenses/Apache-2.0.txt`.
The local pipeline releases completed conditioning intermediates before decode;
this changes buffer lifetime without changing model arithmetic or weights.

The model checkpoint is downloaded from
`Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3`; its weights and
the upstream LTX components retain their own licenses and are not included in
this custom node. The tested Hugging Face revision is
`4655337474c1c77fb52e9ac55111552e625e779c`.
