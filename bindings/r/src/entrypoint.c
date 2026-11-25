// Forward R's package initialization hook to the extendr entry point in Rust.
void R_init_kbolt_extendr(void *dll);

void R_init_kbolt(void *dll) {
    R_init_kbolt_extendr(dll);
}
