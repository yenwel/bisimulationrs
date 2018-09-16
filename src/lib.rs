extern crate carboxyl;


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
