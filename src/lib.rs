extern crate carboxyl;

//to do using reactive streams make bisimulation algorithm
//https://arxiv.org/pdf/1311.7635.pdf
//http://www.math.unipd.it/~crafa/Pubblicazioni/CrafaRanzatoICALP11.pdf
//https://arxiv.org/pdf/1705.08362.pdf

trait State {

}

struct LabelledTransitionSystem {
   states : Vec<String>,
   transitions : Vec<(String,String)>,
   labeling_function : fn((String,String)) -> String
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
