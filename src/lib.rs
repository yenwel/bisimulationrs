extern crate carboxyl;

//to do using reactive streams make bisimulation algorithm
//https://arxiv.org/pdf/1311.7635.pdf
//http://www.math.unipd.it/~crafa/Pubblicazioni/CrafaRanzatoICALP11.pdf
//https://arxiv.org/pdf/1705.08362.pdf

trait State {

}

struct LabelledTransitionSystem<T : State> {
   states : Vec<T>,
   transitions : Vec<(T,T)>,
   labeling_function : fn((T,T)) -> String
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
