use crate::solver::{idx_p, idx_u, idx_v, BenchmarkSummary, TimestepDiagnostic};
use csv::WriterBuilder;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::Path;

pub fn write_summary(path: &Path, summary: &BenchmarkSummary) -> csv::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let exists = path.exists() && path.metadata().map(|m| m.len() > 0).unwrap_or(false);
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = WriterBuilder::new().has_headers(!exists).from_writer(file);
    writer.serialize(summary)?;
    writer.flush()?;
    Ok(())
}

pub fn write_timestep_history(path: &Path, history: &[TimestepDiagnostic]) -> csv::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);
    for row in history {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_fields(
    prefix: &Path,
    nx: usize,
    ny: usize,
    u: &[f64],
    v: &[f64],
    p: &[f64],
) -> io::Result<()> {
    if let Some(parent) = prefix.parent() {
        fs::create_dir_all(parent)?;
    }
    write_u_field(&prefix.with_extension("u.csv"), nx, ny, u)?;
    write_v_field(&prefix.with_extension("v.csv"), nx, ny, v)?;
    write_p_field(&prefix.with_extension("p.csv"), nx, ny, p)?;
    Ok(())
}

pub fn write_validation_fields(
    fields_dir: &Path,
    language: &str,
    run_label: &str,
    tolerance_strategy: &str,
    nx: usize,
    ny: usize,
    re: f64,
    dt: f64,
    nt: usize,
    u: &[f64],
    v: &[f64],
    p: &[f64],
) -> io::Result<()> {
    fs::create_dir_all(fields_dir)?;
    let re_tag = re_tag(re);
    let label_suffix = if run_label.is_empty() {
        String::new()
    } else {
        format!("_{}", sanitize_tag(run_label))
    };
    let stem = format!("{language}_mac_N{nx}_Re{re_tag}{label_suffix}");
    write_u_field(&fields_dir.join(format!("{stem}_u.csv")), nx, ny, u)?;
    write_v_field(&fields_dir.join(format!("{stem}_v.csv")), nx, ny, v)?;
    write_p_field(&fields_dir.join(format!("{stem}_p.csv")), nx, ny, p)?;
    let metadata = format!(
        concat!(
            "{{\n",
            "  \"language\": \"{language}\",\n",
            "  \"run_label\": \"{run_label}\",\n",
            "  \"tolerance_strategy\": \"{tolerance_strategy}\",\n",
            "  \"nx\": {nx},\n",
            "  \"ny\": {ny},\n",
            "  \"Re\": {re:.12},\n",
            "  \"dt\": {dt:.17e},\n",
            "  \"nt\": {nt},\n",
            "  \"final_time\": {final_time:.17e},\n",
            "  \"lid_velocity\": 1.0,\n",
            "  \"viscosity\": {viscosity:.17e},\n",
            "  \"dx\": {dx:.17e},\n",
            "  \"dy\": {dy:.17e},\n",
            "  \"staggering\": \"MAC: p[ny,nx] cell centers; u[ny,nx+1] vertical faces; v[ny+1,nx] horizontal faces; row-major CSV rows are y-index j\"\n",
            "}}\n"
        ),
        language = language,
        run_label = run_label,
        tolerance_strategy = tolerance_strategy,
        nx = nx,
        ny = ny,
        re = re,
        dt = dt,
        nt = nt,
        final_time = dt * nt as f64,
        viscosity = 1.0 / re,
        dx = 1.0 / nx as f64,
        dy = 1.0 / ny as f64,
    );
    fs::write(fields_dir.join(format!("{stem}_metadata.json")), metadata)
}

fn sanitize_tag(tag: &str) -> String {
    tag.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn re_tag(re: f64) -> String {
    if (re - re.round()).abs() < 1.0e-9 {
        format!("{}", re.round() as i64)
    } else {
        format!("{re:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn write_p_field(path: &Path, nx: usize, ny: usize, field: &[f64]) -> io::Result<()> {
    let mut text = String::with_capacity(nx * ny * 14);
    for j in 0..ny {
        for i in 0..nx {
            if i > 0 {
                text.push(',');
            }
            text.push_str(&format!("{:.12e}", field[idx_p(i, j, nx)]));
        }
        text.push('\n');
    }
    fs::write(path, text)
}

fn write_u_field(path: &Path, nx: usize, ny: usize, field: &[f64]) -> io::Result<()> {
    let mut text = String::with_capacity((nx + 1) * ny * 14);
    for j in 0..ny {
        for i in 0..=nx {
            if i > 0 {
                text.push(',');
            }
            text.push_str(&format!("{:.12e}", field[idx_u(i, j, nx)]));
        }
        text.push('\n');
    }
    fs::write(path, text)
}

fn write_v_field(path: &Path, nx: usize, ny: usize, field: &[f64]) -> io::Result<()> {
    let mut text = String::with_capacity(nx * (ny + 1) * 14);
    for j in 0..=ny {
        for i in 0..nx {
            if i > 0 {
                text.push(',');
            }
            text.push_str(&format!("{:.12e}", field[idx_v(i, j, nx)]));
        }
        text.push('\n');
    }
    fs::write(path, text)
}
