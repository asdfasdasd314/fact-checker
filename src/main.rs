use scraper::Html;
use reqwest;

const FBI_VAULT_URL: &str = "https://vault.fbi.gov/";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let body = reqwest::blocking::get(FBI_VAULT_URL)?.text()?;
    let document = Html::parse_document(&body);
    println!("{:#?}", document.html());
    Ok(())
}
